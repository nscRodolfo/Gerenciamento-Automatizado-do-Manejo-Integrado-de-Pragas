package my.aplication.manejointeligentedepragas;

import android.content.Intent;
import android.content.SharedPreferences;

//import android.support.annotation.NonNull;
import androidx.annotation.NonNull;
//import android.support.design.widget.NavigationView;
import com.google.android.material.navigation.NavigationView;
//import android.support.v4.view.GravityCompat;
import androidx.core.view.GravityCompat;
//import android.support.v4.widget.DrawerLayout;
import androidx.drawerlayout.widget.DrawerLayout;
//import android.support.v7.app.ActionBarDrawerToggle;
import androidx.appcompat.app.ActionBarDrawerToggle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
//import android.support.v7.widget.Toolbar;
import androidx.appcompat.widget.Toolbar;

import android.os.Bundle;
import android.view.MenuItem;
import android.view.View;
import android.widget.TextView;
import android.widget.Toast;

import com.example.manejointeligentedepragas.R;

import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;


public class Sobre extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener{

    private DrawerLayout drawerLayout;
    String tipoUsu;
    TextView tv1;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_sobre);

        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_sobre);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_sobre);
        NavigationView navigationView = findViewById(R.id.nav_view_sobre);
        navigationView.setNavigationItemSelectedListener(this);
        ActionBarDrawerToggle toggle = new ActionBarDrawerToggle(this, drawerLayout, toolbar, R.string.navigation_drawer_open, R.string.navigation_drawer_close);
        drawerLayout.addDrawerListener(toggle);
        toggle.syncState();
        View headerView = navigationView.getHeaderView(0);

        Controller_Usuario controller_usuario = new Controller_Usuario(getBaseContext());
        String nomeUsu = controller_usuario.getUser().getNome();
        String emailUsu = controller_usuario.getUser().getEmail();

        TextView nomeMenu = headerView.findViewById(R.id.nomeMenu);
        nomeMenu.setText(nomeUsu);

        TextView emailMenu = headerView.findViewById(R.id.emailMenu);
        emailMenu.setText(emailUsu);


        tv1 = findViewById(R.id.tv1);
        tv1.setText("MIP² é um aplicativo desenvolvido durante um projeto de Iniciação Científica no IFSEMG - campus Rio Pomba em parceria com CNPq, como o objetivo auxiliar o produtor no manejo das pragas utilizando técnicas do Manejo Integrado de Pragas.\n" +
                "\n" +
                "Equipe da IC (alunos) e curso:\n" +
                "Rodolfo Chagas Marinho Nascimento - Ciência da Computação \n" +
                "Monica Simão Giponi - Agroecologia\n" +
                "\n" +
                "Deixamos aqui um agradecimento especial ao nosso grande amigo Daniel Pimenta, que esteve nos ajudando desde o início do projeto. " +
                "\n\nTemos um site de apoio, onde é possível visualizar as informações de maneira mais simples e baixar relatórios em COPY, CSV, EXCEL e PDF. Confira no site: mip.software \n\nPara entrar em contato com qualquer dúvidas, erros e afins a nossa caixa de e-mail está aberta para ajudá-los: agroecomp.corp@gmail.com\n");

    }

    @Override
    public void onBackPressed() {
        if(drawerLayout.isDrawerOpen(GravityCompat.START)){
            drawerLayout.closeDrawer(GravityCompat.START);
        }else {
            super.onBackPressed();
        }
    }

    @Override
    public boolean onNavigationItemSelected(@NonNull MenuItem menuItem) {
        switch (menuItem.getItemId()){
            case R.id.drawerPerfil:
                Controller_Usuario cu = new Controller_Usuario(getBaseContext());
                tipoUsu = cu.getUser().getTipo();
                if(tipoUsu == null){
                    Toast.makeText(Sobre.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                }else{
                    Intent i= new Intent(this, Perfil.class);
                    startActivity(i);
                }
                break;
            case R.id.drawerProp:
                Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
                tipoUsu = cu1.getUser().getTipo();
                if(tipoUsu==null){
                    Toast.makeText(Sobre.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
                }else{
                    Intent prop= new Intent(this, Propriedades.class);
                    startActivity(prop);
                }
                break;
            case R.id.drawerPlantas:
                Intent j = new Intent(this, VisualizaPlantas.class);
                startActivity(j);
                break;

            case R.id.drawerPrag:
                Intent k = new Intent(this, VisualizaPragas.class);
                startActivity(k);
                break;

            case R.id.drawerMet:
                Intent l = new Intent(this, VisualizaMetodos.class);
                startActivity(l);
                break;

            case R.id.drawerSobreMip:
                Intent p = new Intent(this, SobreMIP.class);
                startActivity(p);
                break;

            case R.id.drawerTutorial:
                SharedPreferences pref = getApplicationContext().getSharedPreferences("myPrefs",MODE_PRIVATE);
                SharedPreferences.Editor editor = pref.edit();
                editor.putBoolean("isIntroOpened",false);
                editor.commit();

                Intent intro = new Intent(this, IntroActivity.class);
                startActivity(intro);
                break;

            case R.id.drawerSobre:
                Intent pp = new Intent(this, Sobre.class);
                startActivity(pp);
                break;
            case R.id.drawerReferencias:
                Intent pi = new Intent(this, Referencias.class);
                startActivity(pi);
                break;

            case R.id.drawerRecomendações:
                Intent pa = new Intent(this, RecomendacoesMAPA.class);
                startActivity(pa);
                break;
        }
        drawerLayout.closeDrawer(GravityCompat.START);
        return true;
    }
}
