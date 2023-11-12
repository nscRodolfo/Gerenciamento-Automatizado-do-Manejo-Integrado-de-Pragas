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

import my.aplication.manejointeligentedepragas.Crontroller.Controller_Usuario;

import com.example.manejointeligentedepragas.R;
import com.github.barteksc.pdfviewer.PDFView;

public class RecomendacoesMAPA extends AppCompatActivity implements NavigationView.OnNavigationItemSelectedListener
        {

    PDFView mPDFView;
            private DrawerLayout drawerLayout;
            String tipoUsu;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_recomendacoes_map);


        mPDFView = findViewById(R.id.pdfView);
        mPDFView.fromAsset("orientacoes.pdf").load();


        //menu novo
        Toolbar toolbar = findViewById(R.id.toolbar_Recomendacoes);
        setSupportActionBar(toolbar);
        drawerLayout= findViewById(R.id.drawer_layout_Recomendacoes);
        NavigationView navigationView = findViewById(R.id.nav_view_Recomendacoes);
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
                            Toast.makeText(RecomendacoesMAPA.this,"Para acessar seu perfil, faça login!", Toast.LENGTH_LONG).show();
                        }else{
                            Intent i= new Intent(this, Perfil.class);
                            startActivity(i);
                        }
                        break;
                    case R.id.drawerProp:
                        Controller_Usuario cu1 = new Controller_Usuario(getBaseContext());
                        tipoUsu = cu1.getUser().getTipo();
                        if(tipoUsu==null){
                            Toast.makeText(RecomendacoesMAPA.this,"Para acessar as propriedades, faça login!", Toast.LENGTH_LONG).show();
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
